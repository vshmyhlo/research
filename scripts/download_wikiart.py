import asyncio
import json
import os
import urllib.parse

import aiofiles
import aiohttp
import click


class Client:
    def __init__(self, sess: aiohttp.ClientSession):
        self.sess = sess

    async def get_json(self, path: str):
        url = f"https://www.wikiart.org{path}"
        print(url)
        async with self.sess.get(url) as res:
            if res.content_type != "application/json":
                return None
            return await res.json()

    async def login(self, access_key, secret_key):
        cache_path = "./data/session.json"
        if not os.path.exists(cache_path):
            path = with_query(
                "/en/Api/2/login".format(access_key, secret_key),
                {
                    "accessCode": access_key,
                    "secretCode": secret_key,
                },
            )
            res = await self.get_json(path)
            async with aiofiles.open(cache_path, "w") as f:
                await f.write(json.dumps(res))

        async with aiofiles.open(cache_path) as f:
            res = json.loads(await f.read())

        return res["SessionKey"]

    async def paintings_by_genre(self, genre: str):
        page = 1
        while True:
            path = with_query(
                "/en/paintings-by-genre/{}".format(genre),
                {"json": "2", "page": str(page)},
            )
            res = await self.get_json(path)
            if res is None:
                break
            print("fetched {}, page {}".format(genre, page))
            paintings = res.get("Paintings", []) or []
            for p in paintings:
                yield p
            page += 1

    async def download_image(self, paint, output_path: str, sem: asyncio.Semaphore):
        if paint["image"] == "https://uploads.wikiart.org/Content/images/FRAME-600x480.jpg":
            print("skipping {}".format(paint["id"]))
            return

        _, ext = os.path.splitext(paint["image"])
        image_path = os.path.join(output_path, paint["id"] + ext)
        if os.path.exists(image_path):
            return
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        async with sem, self.sess.get(paint["image"]) as res, aiofiles.open(image_path, "wb") as f:
            while True:
                chunk = await res.content.read(1024)
                if not chunk:
                    break
                await f.write(chunk)

        print('fetched "{}"'.format(paint["title"]))


@click.command()
@click.option("--genre", "-g", type=click.STRING, multiple=True, required=True)
# @click.option("--access-key", type=click.STRING, required=True)
# @click.option("--secret-key", type=click.STRING, required=True)
def main(genre):
    asyncio.run(main_async(genres=genre))


async def main_async(genres):
    output_path = os.path.join("./data/wikiart")
    sem = asyncio.Semaphore(32)
    # tasks = []
    for genre in genres:
        # tasks.append(asyncio.create_task(download_genre(genre, output_path, sem)))
        await download_genre(genre, output_path, sem)
    # for task in tasks:
    #     await task


async def download_genre(genre, output_path, sem):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout()) as sess:
        client = Client(sess)
        tasks = []
        async for paint in client.paintings_by_genre(genre):
            tasks.append(asyncio.create_task(client.download_image(paint, output_path, sem)))
        for task in tasks:
            await task


def with_query(base, query):
    return f"{base}?{urllib.parse.urlencode(query)}"


# async def search_paintings(sess, session_key, search_term, page=None):
#     has_more = True
#     while has_more:
#         query = {"authSessionKey": session_key, "term": search_term, "imageFormat": "Large"}
#         if page is not None:
#             query["paginationToken"] = page
#         url = "en/api/2/PaintingSearch?{}".format(urllib.parse.urlencode(query))
#         print(url)
#         res = await wikiart_get_json(sess, url)
#         if "data" not in res:
#             print(res)
#         for paint in res["data"]:
#             yield paint
#         print(res["hasMore"], res["paginationToken"])
#
#         # has_more = res["hasMore"]
#         page = res["paginationToken"]
#
#         await asyncio.sleep(10)


# class WikiArtClient:
#     def __init__(self, sess, session_key):
#         self.sess = sess
#         self.session_key = session_key
#
#     async def search_paintings(self, query):
#         url = "en/api/2/PaintingSearch?authSessionKey={}&term={}&paginationToken={}&imageFormat={}".format(
#             self.session_key, query, "", "Large"
#         )
#         res = await wikiart_get_json(self.sess, url)
#         return res["data"]
#
#     async def painting_details(self, id):
#         url = "en/api/2/Painting?authSessionKey={}&id={}&imageFormat={}".format(
#             self.session_key, id, "Large"
#         )
#         return await wikiart_get_json(self.sess, url)
#

if __name__ == "__main__":
    main()
